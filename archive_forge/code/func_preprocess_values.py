import os
from editorconfig import VERSION
from editorconfig.exceptions import PathError, VersionError
from editorconfig.ini import EditorConfigParser
def preprocess_values(self):
    """Preprocess option values for consumption by plugins"""
    opts = self.options
    for name in ['end_of_line', 'indent_style', 'indent_size', 'insert_final_newline', 'trim_trailing_whitespace', 'charset']:
        if name in opts:
            opts[name] = opts[name].lower()
    if opts.get('indent_style') == 'tab' and (not 'indent_size' in opts) and (self.version >= (0, 10, 0)):
        opts['indent_size'] = 'tab'
    if 'indent_size' in opts and 'tab_width' not in opts and (opts['indent_size'] != 'tab'):
        opts['tab_width'] = opts['indent_size']
    if 'indent_size' in opts and 'tab_width' in opts and (opts['indent_size'] == 'tab'):
        opts['indent_size'] = opts['tab_width']