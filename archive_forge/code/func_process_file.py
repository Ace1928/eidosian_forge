import argparse
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import ipynb
from tensorflow.tools.compatibility import tf_upgrade_v2
from tensorflow.tools.compatibility import tf_upgrade_v2_safety
def process_file(in_filename, out_filename, upgrader):
    """Process a file of type `.py` or `.ipynb`."""
    if in_filename.endswith('.py'):
        files_processed, report_text, errors = upgrader.process_file(in_filename, out_filename)
    elif in_filename.endswith('.ipynb'):
        files_processed, report_text, errors = ipynb.process_file(in_filename, out_filename, upgrader)
    else:
        raise NotImplementedError('Currently converter only supports python or ipynb')
    return (files_processed, report_text, errors)