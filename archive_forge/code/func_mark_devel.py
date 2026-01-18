from __future__ import absolute_import, division, print_function
def mark_devel(self, var):
    result = var.split('-')[0] + '-devel'
    return result