import logging
import re
def make_key_function(loose):

    def key_function(version):
        v = make_semver(version, loose)
        key = (v.major, v.minor, v.patch)
        if v.prerelease:
            key = key + tuple(v.prerelease)
        else:
            key = (*key, float('inf'))
        return key
    return key_function