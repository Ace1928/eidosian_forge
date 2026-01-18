import argparse
import copy
import re
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import evaluator
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import source_utils
def print_tensor(self, args, screen_info=None):
    """Command handler for print_tensor.

    Print value of a given dumped tensor.

    Args:
      args: Command-line arguments, excluding the command prefix, as a list of
        str.
      screen_info: Optional dict input containing screen information such as
        cols.

    Returns:
      Output text lines as a RichTextLines object.
    """
    parsed = self._arg_parsers['print_tensor'].parse_args(args)
    np_printoptions = cli_shared.numpy_printoptions_from_screen_info(screen_info)
    highlight_options = cli_shared.parse_ranges_highlight(parsed.ranges)
    tensor_name, tensor_slicing = command_parser.parse_tensor_name_with_slicing(parsed.tensor_name)
    node_name, output_slot = debug_graphs.parse_node_or_tensor_name(tensor_name)
    if self._debug_dump.loaded_partition_graphs() and (not self._debug_dump.node_exists(node_name)):
        output = cli_shared.error('Node "%s" does not exist in partition graphs' % node_name)
        _add_main_menu(output, node_name=None, enable_list_tensors=True, enable_print_tensor=False)
        return output
    watch_keys = self._debug_dump.debug_watch_keys(node_name)
    if output_slot is None:
        output_slots = set()
        for watch_key in watch_keys:
            output_slots.add(int(watch_key.split(':')[1]))
        if len(output_slots) == 1:
            output_slot = list(output_slots)[0]
        else:
            lines = ['Node "%s" generated debug dumps from %s output slots:' % (node_name, len(output_slots)), 'Please specify the output slot: %s:x.' % node_name]
            output = debugger_cli_common.RichTextLines(lines)
            _add_main_menu(output, node_name=node_name, enable_list_tensors=True, enable_print_tensor=False)
            return output
    matching_data = []
    for watch_key in watch_keys:
        debug_tensor_data = self._debug_dump.watch_key_to_data(watch_key)
        for datum in debug_tensor_data:
            if datum.output_slot == output_slot:
                matching_data.append(datum)
    if not matching_data:
        output = cli_shared.error('Tensor "%s" did not generate any dumps.' % parsed.tensor_name)
    elif len(matching_data) == 1:
        if parsed.number <= 0:
            output = cli_shared.format_tensor(matching_data[0].get_tensor(), matching_data[0].watch_key, np_printoptions, print_all=parsed.print_all, tensor_slicing=tensor_slicing, highlight_options=highlight_options, include_numeric_summary=parsed.numeric_summary, write_path=parsed.write_path)
        else:
            output = cli_shared.error('Invalid number (%d) for tensor %s, which generated one dump.' % (parsed.number, parsed.tensor_name))
        _add_main_menu(output, node_name=node_name, enable_print_tensor=False)
    else:
        if parsed.number < 0:
            lines = ['Tensor "%s" generated %d dumps:' % (parsed.tensor_name, len(matching_data))]
            font_attr_segs = {}
            for i, datum in enumerate(matching_data):
                rel_time = (datum.timestamp - self._debug_dump.t0) / 1000.0
                lines.append('#%d [%.3f ms] %s' % (i, rel_time, datum.watch_key))
                command = 'print_tensor %s -n %d' % (parsed.tensor_name, i)
                font_attr_segs[len(lines) - 1] = [(len(lines[-1]) - len(datum.watch_key), len(lines[-1]), debugger_cli_common.MenuItem(None, command))]
            lines.append('')
            lines.append('You can use the -n (--number) flag to specify which dump to print.')
            lines.append('For example:')
            lines.append('  print_tensor %s -n 0' % parsed.tensor_name)
            output = debugger_cli_common.RichTextLines(lines, font_attr_segs=font_attr_segs)
        elif parsed.number >= len(matching_data):
            output = cli_shared.error('Specified number (%d) exceeds the number of available dumps (%d) for tensor %s' % (parsed.number, len(matching_data), parsed.tensor_name))
        else:
            output = cli_shared.format_tensor(matching_data[parsed.number].get_tensor(), matching_data[parsed.number].watch_key + ' (dump #%d)' % parsed.number, np_printoptions, print_all=parsed.print_all, tensor_slicing=tensor_slicing, highlight_options=highlight_options, write_path=parsed.write_path)
        _add_main_menu(output, node_name=node_name, enable_print_tensor=False)
    return output