from distutils.core import Distribution
def has_scons_scripts(self):
    return bool(self.scons_data)