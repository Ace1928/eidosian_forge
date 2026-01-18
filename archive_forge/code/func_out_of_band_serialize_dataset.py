import contextlib
import traceback
import ray
@contextlib.contextmanager
def out_of_band_serialize_dataset():
    context = ray._private.worker.global_worker.get_serialization_context()
    try:
        context._register_cloudpickle_reducer(ray.data.Dataset, _reduce)
        yield
    finally:
        context._unregister_cloudpickle_reducer(ray.data.Dataset)