from Cryptodome.Util._raw_api import load_pycryptodome_raw_lib
def have_aes_ni():
    return _raw_cpuid_lib.have_aes_ni()