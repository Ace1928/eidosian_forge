from collections.abc import Mapping
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def make_filler_dataset(ds):
    padding = cardinality - ds.cardinality()
    filler_element = nest.map_structure(lambda spec: array_ops.zeros(spec.shape, spec.dtype), ds.element_spec)
    filler_element[mask_key] = False
    filler_dataset = dataset_ops.Dataset.from_tensors(filler_element)
    filler_dataset = filler_dataset.repeat(padding)
    return filler_dataset