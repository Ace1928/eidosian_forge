import struct
import logging
@staticmethod
def register_sflow_version(version):

    def _register_sflow_version(cls):
        sFlow._SFLOW_VERSIONS[version] = cls
        return cls
    return _register_sflow_version