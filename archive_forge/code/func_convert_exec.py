import json
import sys
import argparse
from pyomo.common.collections import Bunch
from pyomo.opt import guess_format
from pyomo.scripting.pyomo_parser import add_subparser, CustomHelpFormatter
from pyomo.scripting.solve_config import Default_Config
def convert_exec(args, unparsed):
    import pyomo.scripting.util
    if not args.template is None:
        config, blocks = Default_Config().config_block(True)
        OUTPUT = open(args.template, 'w')
        if args.template.endswith('json'):
            OUTPUT.write(json.dumps(config.value(), indent=2))
        else:
            OUTPUT.write(config.generate_yaml_template())
        OUTPUT.close()
        print("  Created template file '%s'" % args.template)
        sys.exit(0)
    save_filename = getattr(args, 'filename', None)
    if save_filename is None:
        save_format = getattr(args, 'format', None)
        if not save_format is None:
            save_filename = 'unknown.' + save_format
    if save_filename is None:
        try:
            val = pyomo.scripting.util.get_config_values(unparsed[-1])
        except IndexError:
            val = None
        except IOError:
            val = None
        if not val is None:
            try:
                save_filename = val['model']['save file']
            except:
                pass
            if save_filename is None:
                try:
                    save_filename = 'unknown.' + str(val['model']['save format'])
                except:
                    pass
    if save_filename is None or '-h' in unparsed or '--help' in unparsed:
        if not ('-h' in unparsed or '--help' in unparsed):
            print('ERROR: No output file or format specified!')
            print('')
        config, blocks = Default_Config().config_block()
        parser = create_temporary_parser(output=True, generate=True)
        config.initialize_argparse(parser)
        parser.parse_args(args=unparsed + ['-h'])
        sys.exit(1)
    config, blocks = Default_Config().config_block()
    _parser = create_temporary_parser()
    config.initialize_argparse(_parser)
    _options = _parser.parse_args(args=unparsed)
    config.import_argparse(_options)
    config.model.save_file = getattr(args, 'filename', None)
    config.model.save_format = getattr(args, 'format', None)
    if _options.model_or_config_file.endswith('.py'):
        config.model.filename = _options.model_or_config_file
        config.data.files = _options.data_files
    else:
        val = pyomo.scripting.util.get_config_values(_options.model_or_config_file)
        config.set_value(val)
    return pyomo.scripting.util.run_command(command=run_convert, parser=convert_parser, options=config, name='convert')