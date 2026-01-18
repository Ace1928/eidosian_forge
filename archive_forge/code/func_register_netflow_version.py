import struct
@staticmethod
def register_netflow_version(version):

    def _register_netflow_version(cls):
        NetFlow._NETFLOW_VERSIONS[version] = cls
        return cls
    return _register_netflow_version