import argparse
import ast
import re
import sys
def extract_output_file_path(args):
    """Extract output file path from command arguments.

  Args:
    args: (list of str) command arguments.

  Returns:
    (list of str) Command arguments with the output file path part stripped.
    (str or None) Output file path (if any).

  Raises:
    SyntaxError: If there is no file path after the last ">" character.
  """
    if args and args[-1].endswith('>'):
        raise SyntaxError('Redirect file path is empty')
    elif args and args[-1].startswith('>'):
        try:
            _parse_interval(args[-1])
            if len(args) > 1 and args[-2].startswith('-'):
                output_file_path = None
            else:
                output_file_path = args[-1][1:]
                args = args[:-1]
        except ValueError:
            output_file_path = args[-1][1:]
            args = args[:-1]
    elif len(args) > 1 and args[-2] == '>':
        output_file_path = args[-1]
        args = args[:-2]
    elif args and args[-1].count('>') == 1:
        gt_index = args[-1].index('>')
        if gt_index > 0 and args[-1][gt_index - 1] == '=':
            output_file_path = None
        else:
            output_file_path = args[-1][gt_index + 1:]
            args[-1] = args[-1][:gt_index]
    elif len(args) > 1 and args[-2].endswith('>'):
        output_file_path = args[-1]
        args = args[:-1]
        args[-1] = args[-1][:-1]
    else:
        output_file_path = None
    return (args, output_file_path)