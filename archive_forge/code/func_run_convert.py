import json
import sys
import argparse
from pyomo.common.collections import Bunch
from pyomo.opt import guess_format
from pyomo.scripting.pyomo_parser import add_subparser, CustomHelpFormatter
from pyomo.scripting.solve_config import Default_Config
def run_convert(options=Bunch(), parser=None):
    from pyomo.scripting.convert import convert, convert_dakota
    if options.model.save_format is None and options.model.save_file:
        options.model.save_format = options.model.save_file.split('.')[-1]
    _format = guess_format(options.model.save_format)
    if options.model.save_format == 'dakota':
        return convert_dakota(options, parser)
    elif _format is None:
        if options.model.save_format is None:
            raise RuntimeError('Unspecified target conversion format!')
        else:
            raise RuntimeError('Unrecognized target conversion format (%s)!' % (options.model.save_format,))
    else:
        return convert(options, parser, _format)