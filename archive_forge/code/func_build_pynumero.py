import sys
from pyomo.common.cmake_builder import build_cmake_project
def build_pynumero(user_args=[], parallel=None):
    return build_cmake_project(targets=['src'], package_name='pynumero_libraries', description='PyNumero libraries', user_args=['-DBUILD_AMPLASL_IF_NEEDED=ON'] + user_args, parallel=parallel)