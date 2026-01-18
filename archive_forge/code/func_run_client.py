import os
import sys
from absl import flags
import portpicker
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.python.platform import test as tf_test
def run_client(idx, num_clients, server_ports, additional_ports, num_local_devices, client_config_function):
    """Runs test.main() from a DTensor Client process on localhost.

  This function runs in a separate process so that the eager context is
  properly separated, which resembles real world multi-client setup.

  Virtual devices are configured before test.main() is called.

  Each client is configured to only have access to the physical GPU device
  corresponding to its client id via CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES.

  Each client is configured to only have access to some TPU cores
  corresponding to its client id via flags.

  The clients redirect stdout and stderr to files under Test Artifacts.

  Args:
    idx: integer task number represents the client's id from global picture.
    num_clients: total number of clients.
    server_ports: A list of ports that is allocated and to be used to construct
      GRPC server. server_ports[idx] will be the GRPC server on the
      corresponding client.
    additional_ports: A list of ports that is allocated and to be used to
      construct the backends.
    num_local_devices: Number of devices per client.
    client_config_function: A function, for each of the client to config the
      local environment variables, etc. Note that the function will be called
      with a dict of extra params, eg:
        {'num_clients': 2
         'client_id': 0,
         'worker_jobs': ['localhost:port1', 'localhost:port2'],
         'num_devices': 4,
        }
  """
    test_backend_util.slice_host_devices_for_multiworker(num_clients, idx, additional_ports)
    artifact_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', '')
    if artifact_dir:
        with open(os.path.join(artifact_dir, f'test-client-process-{idx}.log'), 'wb') as fp:
            os.dup2(fp.fileno(), 1)
            os.dup2(fp.fileno(), 2)
    worker_jobs = [f'localhost:{port:06d}' for port in server_ports]
    client_config_func_param = {'num_clients': num_clients, 'client_id': idx, 'worker_jobs': worker_jobs, 'num_devices': num_local_devices}
    client_config_function(client_config_func_param)
    tf_test.main()