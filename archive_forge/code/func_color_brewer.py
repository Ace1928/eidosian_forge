import base64
import json
import math
import os
import re
import struct
import typing
import zlib
from typing import Any, Callable, Union
from jinja2 import Environment, PackageLoader
def color_brewer(color_code, n=6):
    """
    Generate a colorbrewer color scheme of length 'len', type 'scheme.
    Live examples can be seen at http://colorbrewer2.org/

    """
    maximum_n = 253
    minimum_n = 3
    if not isinstance(n, int):
        raise TypeError('n has to be an int, not a %s' % type(n))
    if n > maximum_n:
        raise ValueError('The maximum number of colors in a ColorBrewer sequential color series is 253')
    if n < minimum_n:
        raise ValueError('The minimum number of colors in a ColorBrewer sequential color series is 3')
    if not isinstance(color_code, str):
        raise ValueError(f'color should be a string, not a {type(color_code)}.')
    if color_code[-2:] == '_r':
        base_code = color_code[:-2]
        core_color_code = base_code + '_' + str(n).zfill(2)
        color_reverse = True
    else:
        base_code = color_code
        core_color_code = base_code + '_' + str(n).zfill(2)
        color_reverse = False
    with open(os.path.join(rootpath, '_schemes.json')) as f:
        schemes = json.loads(f.read())
    with open(os.path.join(rootpath, 'scheme_info.json')) as f:
        scheme_info = json.loads(f.read())
    with open(os.path.join(rootpath, 'scheme_base_codes.json')) as f:
        core_schemes = json.loads(f.read())['codes']
    if base_code not in core_schemes:
        raise ValueError(base_code + ' is not a valid ColorBrewer code')
    explicit_scheme = True
    if schemes.get(core_color_code) is None:
        explicit_scheme = False
    if not explicit_scheme:
        if scheme_info[base_code] == 'Qualitative':
            matching_quals = []
            for key in schemes:
                if base_code + '_' in key:
                    matching_quals.append(int(key.split('_')[1]))
            raise ValueError('Expanded color support is not available for Qualitative schemes; restrict the number of colors for the ' + base_code + ' code to between ' + str(min(matching_quals)) + ' and ' + str(max(matching_quals)))
        else:
            longest_scheme_name = base_code
            longest_scheme_n = 0
            for sn_name in schemes.keys():
                if '_' not in sn_name:
                    continue
                if sn_name.split('_')[0] != base_code:
                    continue
                if int(sn_name.split('_')[1]) > longest_scheme_n:
                    longest_scheme_name = sn_name
                    longest_scheme_n = int(sn_name.split('_')[1])
            if not color_reverse:
                color_scheme = linear_gradient(schemes.get(longest_scheme_name), n)
            else:
                color_scheme = linear_gradient(schemes.get(longest_scheme_name)[::-1], n)
    elif not color_reverse:
        color_scheme = schemes.get(core_color_code, None)
    else:
        color_scheme = schemes.get(core_color_code, None)[::-1]
    return color_scheme