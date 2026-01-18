import os
import pytest
import ray
@pytest.fixture(scope='session')
def shared_ray_instance():
    if 'RAY_ADDRESS' in os.environ:
        del os.environ['RAY_ADDRESS']
    yield ray.init(num_cpus=16, namespace=TEST_NAMESPACE, log_to_driver=True)