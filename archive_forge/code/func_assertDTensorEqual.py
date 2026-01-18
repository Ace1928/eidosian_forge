import collections
import copy
import itertools
import json
import os
import typing
from absl import flags
from absl.testing import parameterized
import numpy as np
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.config import is_gpu_present  # pylint: disable=unused-import
from tensorflow.dtensor.python.config import is_tpu_present  # pylint: disable=unused-import
from tensorflow.dtensor.python.config import preferred_device_type  # pylint: disable=unused-import
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.dtensor.python.tests.test_backend_name import DTENSOR_TEST_UTIL_BACKEND
from tensorflow.dtensor.python.tests.test_backend_name import DTensorTestUtilBackend
from tensorflow.dtensor.python.tests.test_backend_util import DTensorTestBackendConfigurator
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test as tf_test
def assertDTensorEqual(self, expected_result, expected_layout, result_dtensor, tol=DEFAULT_TOL):
    """Asserts DTensor is of the particular value."""
    if issubclass(type(result_dtensor), resource_variable_ops.BaseResourceVariable):
        result_dtensor = result_dtensor.value()
    if expected_layout is not None:
        expected_str = expected_layout.to_string()
        got_str = api.fetch_layout(result_dtensor).to_string()
        index_for_mesh = expected_str.find('mesh:')
        if index_for_mesh != -1 and got_str.find(expected_str[index_for_mesh:]) != -1:
            expected_str = expected_str[:index_for_mesh]
            got_str = got_str[:got_str.find('mesh:')]
        self.assertEqual(api.fetch_layout(result_dtensor), expected_layout, msg='=======\nexpected layout is\n  {}\n\nwhile got layout is\n  {}\n'.format(expected_str, got_str))
    layout = api.fetch_layout(result_dtensor)
    unpacked = [t.numpy() for t in api.unpack(result_dtensor)]
    self.assertAllEqual(expected_result.shape, result_dtensor.shape)
    result_dtensor = numpy_util.to_numpy(result_dtensor)
    self.assertEqual(expected_result.dtype, result_dtensor.dtype, result_dtensor)
    self.assertAllClose(expected_result, result_dtensor, atol=tol, rtol=tol)

    def hash_key(loc):
        """Hash key for Python dict."""
        d = collections.OrderedDict(sorted(loc.items(), key=lambda x: x[0]))
        return json.dumps(d)
    offset_to_mesh_loc_dict = layout.mesh.unravel_index()
    mesh_loc_to_offset_dict = {}
    for offset, loc in offset_to_mesh_loc_dict.items():
        mesh_loc_to_offset_dict[hash_key(loc)] = offset
    replicated_dims = [x for x in layout.mesh.dim_names if x not in layout.sharding_specs]
    for offset, tensor in enumerate(unpacked):
        mesh_loc = offset_to_mesh_loc_dict[offset]
        for dim_sharding in replicated_dims:
            if mesh_loc[dim_sharding] != 0:
                mesh_loc = copy.deepcopy(mesh_loc)
                mesh_loc[dim_sharding] = 0
                offset = mesh_loc_to_offset_dict[hash_key(mesh_loc)]
                self.assertAllClose(tensor, unpacked[offset])