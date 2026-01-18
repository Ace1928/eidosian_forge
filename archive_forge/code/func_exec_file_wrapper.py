import sys
def exec_file_wrapper(fpath, g_vars, l_vars):
    execfile(fpath, g_vars, l_vars)