import types
from keras.src.distribution import distribution_lib
from keras.src.trainers.data_adapters import array_data_adapter
from keras.src.trainers.data_adapters import py_dataset_adapter
from keras.src.trainers.data_adapters.array_data_adapter import ArrayDataAdapter
from keras.src.trainers.data_adapters.generator_data_adapter import (
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDatasetAdapter
from keras.src.trainers.data_adapters.tf_dataset_adapter import TFDatasetAdapter
from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
def get_data_adapter(x, y=None, sample_weight=None, batch_size=None, steps_per_epoch=None, shuffle=False, class_weight=None):
    distribution = distribution_lib.distribution()
    if getattr(distribution, '_is_multi_process', False) and (not is_tf_dataset(x)):
        raise ValueError(f'When using multi-worker distribution, the data must be provided as a `tf.data.Dataset` instance. Received: type(x)={type(x)}.')
    if array_data_adapter.can_convert_arrays((x, y, sample_weight)):
        return ArrayDataAdapter(x, y, sample_weight=sample_weight, class_weight=class_weight, shuffle=shuffle, batch_size=batch_size, steps=steps_per_epoch)
    elif is_tf_dataset(x):
        if y is not None:
            raise_unsupported_arg('y', 'the targets', 'tf.data.Dataset')
        if sample_weight is not None:
            raise_unsupported_arg('sample_weights', 'the sample weights', 'tf.data.Dataset')
        return TFDatasetAdapter(x, class_weight=class_weight, distribution=distribution)
    elif isinstance(x, py_dataset_adapter.PyDataset):
        if y is not None:
            raise_unsupported_arg('y', 'the targets', 'PyDataset')
        if sample_weight is not None:
            raise_unsupported_arg('sample_weights', 'the sample weights', 'PyDataset')
        return PyDatasetAdapter(x, class_weight=class_weight, shuffle=shuffle)
    elif is_torch_dataloader(x):
        if y is not None:
            raise_unsupported_arg('y', 'the targets', 'torch DataLoader')
        if sample_weight is not None:
            raise_unsupported_arg('sample_weights', 'the sample weights', 'torch DataLoader')
        if class_weight is not None:
            raise ValueError(f'Argument `class_weight` is not supported for torch DataLoader inputs. Received: class_weight={class_weight}')
        return TorchDataLoaderAdapter(x)
    elif isinstance(x, types.GeneratorType):
        if y is not None:
            raise_unsupported_arg('y', 'the targets', 'PyDataset')
        if sample_weight is not None:
            raise_unsupported_arg('sample_weights', 'the sample weights', 'PyDataset')
        if class_weight is not None:
            raise ValueError(f'Argument `class_weight` is not supported for Python generator inputs. Received: class_weight={class_weight}')
        return GeneratorDataAdapter(x)
    else:
        raise ValueError(f'Unrecognized data type: x={x} (of type {type(x)})')