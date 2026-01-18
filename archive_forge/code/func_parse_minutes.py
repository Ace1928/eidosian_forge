from __future__ import absolute_import, division, print_function
import re
def parse_minutes(module, period):
    try:
        return parse_duration(period)
    except ValueError:
        module.fail_json(msg="'{0}' is not a valid time period, use combination of data units (Y,W,D,H,M)e.g. 4W3D5H, 5D8H5M, 3D, 5W, 1Y5W...".format(period))