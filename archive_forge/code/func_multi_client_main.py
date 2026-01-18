import os
import sys
from absl import flags
import portpicker
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.python.platform import test as tf_test
def multi_client_main(client_config_function):
    """Creates a Flock of TensorFlow Processes on localhost."""
    flags.FLAGS(sys.argv, known_only=True)
    num_clients = _NUM_CLIENTS.value
    num_process = num_clients or 1
    num_local_devices = _NUM_LOCAL_DEVICES.value
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HIP_VISIBLE_DEVICES'] = ''
    mp_context = test_backend_util.get_mp_context()
    print('Check per client log in Test artifacts.', flush=True)
    server_ports = sorted([portpicker.pick_unused_port() for _ in range(num_process)], reverse=True)
    additional_ports = sorted([portpicker.pick_unused_port() for _ in range(num_process)])
    procs = []
    for client_idx in range(num_process):
        proc = mp_context.Process(target=run_client, args=(client_idx, num_clients, server_ports, additional_ports, num_local_devices, client_config_function), name=f'Client-{client_idx}')
        proc.start()
        procs.append(proc)
    exitcode = 0
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            exitcode = proc.exitcode
    sys.exit(exitcode)