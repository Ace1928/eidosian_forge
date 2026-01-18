from collections import OrderedDict
import importlib
def allgather_global_data(self, local_data):
    assert len(local_data) == len(self._local_map)
    if not self._mpi_interface.have_mpi:
        return list(local_data)
    comm = self._mpi_interface.comm
    global_data_list_of_lists = comm.allgather(local_data)
    return self._stack_global_data(global_data_list_of_lists)