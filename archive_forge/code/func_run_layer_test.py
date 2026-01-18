import json
import shutil
import tempfile
import unittest
import numpy as np
import tree
from keras.src import backend
from keras.src import ops
from keras.src import utils
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.global_state import clear_session
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.models import Model
from keras.src.utils import traceback_utils
from keras.src.utils.shape_utils import map_shape_structure
def run_layer_test(self, layer_cls, init_kwargs, input_shape=None, input_dtype='float32', input_sparse=False, input_data=None, call_kwargs=None, expected_output_shape=None, expected_output_dtype=None, expected_output_sparse=False, expected_output=None, expected_num_trainable_weights=None, expected_num_non_trainable_weights=None, expected_num_non_trainable_variables=None, expected_num_seed_generators=None, expected_num_losses=None, supports_masking=None, expected_mask_shape=None, custom_objects=None, run_training_check=True, run_mixed_precision_check=True):
    """Run basic checks on a layer.

        Args:
            layer_cls: The class of the layer to test.
            init_kwargs: Dict of arguments to be used to
                instantiate the layer.
            input_shape: Shape tuple (or list/dict of shape tuples)
                to call the layer on.
            input_dtype: Corresponding input dtype.
            input_sparse: Whether the input is a sparse tensor (this requires
                the backend to support sparse tensors).
            input_data: Tensor (or list/dict of tensors)
                to call the layer on.
            call_kwargs: Dict of arguments to use when calling the
                layer (does not include the first input tensor argument)
            expected_output_shape: Shape tuple
                (or list/dict of shape tuples)
                expected as output.
            expected_output_dtype: dtype expected as output.
            expected_output_sparse: Whether the output is expected to be sparse
                (this requires the backend to support sparse tensors).
            expected_output: Expected output tensor -- only
                to be specified if input_data is provided.
            expected_num_trainable_weights: Expected number
                of trainable weights of the layer once built.
            expected_num_non_trainable_weights: Expected number
                of non-trainable weights of the layer once built.
            expected_num_seed_generators: Expected number of
                SeedGenerators objects of the layer once built.
            expected_num_losses: Expected number of loss tensors
                produced when calling the layer.
            supports_masking: If True, will check that the layer
                supports masking.
            expected_mask_shape: Expected mask shape tuple
                returned by compute_mask() (only supports 1 shape).
            custom_objects: Dict of any custom objects to be
                considered during deserialization.
            run_training_check: Whether to attempt to train the layer
                (if an input shape or input data was provided).
            run_mixed_precision_check: Whether to test the layer with a mixed
                precision dtype policy.
        """
    if input_shape is not None and input_data is not None:
        raise ValueError('input_shape and input_data cannot be passed at the same time.')
    if expected_output_shape is not None and expected_output is not None:
        raise ValueError('expected_output_shape and expected_output cannot be passed at the same time.')
    if expected_output is not None and input_data is None:
        raise ValueError('In order to use expected_output, input_data must be provided.')
    if expected_mask_shape is not None and supports_masking is not True:
        raise ValueError('In order to use expected_mask_shape, supports_masking\n                must be True.')
    init_kwargs = init_kwargs or {}
    call_kwargs = call_kwargs or {}
    layer = layer_cls(**init_kwargs)
    self.run_class_serialization_test(layer, custom_objects)
    if supports_masking is not None:
        self.assertEqual(layer.supports_masking, supports_masking, msg='Unexpected supports_masking value')

    def run_build_asserts(layer):
        self.assertTrue(layer.built)
        if expected_num_trainable_weights is not None:
            self.assertLen(layer.trainable_weights, expected_num_trainable_weights, msg='Unexpected number of trainable_weights')
        if expected_num_non_trainable_weights is not None:
            self.assertLen(layer.non_trainable_weights, expected_num_non_trainable_weights, msg='Unexpected number of non_trainable_weights')
        if expected_num_non_trainable_variables is not None:
            self.assertLen(layer.non_trainable_variables, expected_num_non_trainable_variables, msg='Unexpected number of non_trainable_variables')
        if expected_num_seed_generators is not None:
            self.assertLen(layer._seed_generators, expected_num_seed_generators, msg='Unexpected number of _seed_generators')

    def run_output_asserts(layer, output, eager=False):
        if expected_output_shape is not None:
            if isinstance(expected_output_shape, tuple):
                self.assertEqual(expected_output_shape, output.shape, msg='Unexpected output shape')
            elif isinstance(expected_output_shape, dict):
                self.assertIsInstance(output, dict)
                self.assertEqual(set(output.keys()), set(expected_output_shape.keys()), msg='Unexpected output dict keys')
                output_shape = {k: v.shape for k, v in expected_output_shape.items()}
                self.assertEqual(expected_output_shape, output_shape, msg='Unexpected output shape')
            elif isinstance(expected_output_shape, list):
                self.assertIsInstance(output, list)
                self.assertEqual(len(output), len(expected_output_shape), msg='Unexpected number of outputs')
                output_shape = [v.shape for v in expected_output_shape]
                self.assertEqual(expected_output_shape, output_shape, msg='Unexpected output shape')
        if expected_output_dtype is not None:
            output_dtype = tree.flatten(output)[0].dtype
            self.assertEqual(expected_output_dtype, backend.standardize_dtype(output_dtype), msg='Unexpected output dtype')
        if expected_output_sparse:
            for x in tree.flatten(output):
                if isinstance(x, KerasTensor):
                    self.assertTrue(x.sparse)
                elif backend.backend() == 'tensorflow':
                    import tensorflow as tf
                    self.assertIsInstance(x, tf.SparseTensor)
                elif backend.backend() == 'jax':
                    import jax.experimental.sparse as jax_sparse
                    self.assertIsInstance(x, jax_sparse.JAXSparse)
                else:
                    self.fail(f'Sparse is unsupported with backend {backend.backend()}')
        if eager:
            if expected_output is not None:
                self.assertEqual(type(expected_output), type(output))
                for ref_v, v in zip(tree.flatten(expected_output), tree.flatten(output)):
                    self.assertAllClose(ref_v, v, msg='Unexpected output value')
            if expected_num_losses is not None:
                self.assertLen(layer.losses, expected_num_losses)

    def run_training_step(layer, input_data, output_data):

        class TestModel(Model):

            def __init__(self, layer):
                super().__init__()
                self.layer = layer

            def call(self, x):
                return self.layer(x)
        model = TestModel(layer)
        data = (input_data, output_data)
        if backend.backend() == 'torch':
            data = tree.map_structure(backend.convert_to_numpy, data)

        def data_generator():
            while True:
                yield data
        jit_compile = 'auto'
        if backend.backend() == 'tensorflow' and input_sparse:
            jit_compile = False
        model.compile(optimizer='sgd', loss='mse', jit_compile=jit_compile)
        model.fit(data_generator(), steps_per_epoch=1, verbose=0)
    if input_data is not None or input_shape is not None:
        if input_shape is None:
            build_shape = tree.map_structure(lambda x: ops.shape(x), input_data)
        else:
            build_shape = input_shape
        layer = layer_cls(**init_kwargs)
        if isinstance(build_shape, dict):
            layer.build(**build_shape)
        else:
            layer.build(build_shape)
        run_build_asserts(layer)
        if input_shape is None:
            keras_tensor_inputs = tree.map_structure(lambda x: create_keras_tensors(ops.shape(x), x.dtype, input_sparse), input_data)
        else:
            keras_tensor_inputs = create_keras_tensors(input_shape, input_dtype, input_sparse)
        layer = layer_cls(**init_kwargs)
        if isinstance(keras_tensor_inputs, dict):
            keras_tensor_outputs = layer(**keras_tensor_inputs, **call_kwargs)
        else:
            keras_tensor_outputs = layer(keras_tensor_inputs, **call_kwargs)
        run_build_asserts(layer)
        run_output_asserts(layer, keras_tensor_outputs, eager=False)
        if expected_mask_shape is not None:
            output_mask = layer.compute_mask(keras_tensor_inputs)
            self.assertEqual(expected_mask_shape, output_mask.shape)
    if input_data is not None or input_shape is not None:
        if input_data is None:
            input_data = create_eager_tensors(input_shape, input_dtype, input_sparse)
        layer = layer_cls(**init_kwargs)
        if isinstance(input_data, dict):
            output_data = layer(**input_data, **call_kwargs)
        else:
            output_data = layer(input_data, **call_kwargs)
        run_output_asserts(layer, output_data, eager=True)
        if run_training_check:
            run_training_step(layer, input_data, output_data)
        if run_mixed_precision_check and backend.backend() == 'torch':
            import torch
            run_mixed_precision_check = torch.cuda.is_available()
        if run_mixed_precision_check:
            layer = layer_cls(**{**init_kwargs, 'dtype': 'mixed_float16'})
            if isinstance(input_data, dict):
                output_data = layer(**input_data, **call_kwargs)
            else:
                output_data = layer(input_data, **call_kwargs)
            for tensor in tree.flatten(output_data):
                dtype = standardize_dtype(tensor.dtype)
                if is_float_dtype(dtype):
                    self.assertEqual(dtype, 'float16')
            for weight in layer.weights:
                dtype = standardize_dtype(weight.dtype)
                if is_float_dtype(dtype):
                    self.assertEqual(dtype, 'float32')