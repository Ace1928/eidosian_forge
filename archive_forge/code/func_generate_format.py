import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_format(self):
    """
        Means that value have to be in specified format. For example date, email or other.

        .. code-block:: python

            {'format': 'email'}

        Valid value for this definition is user@example.com but not @username
        """
    if not self._use_formats:
        return
    with self.l('if isinstance({variable}, str):'):
        format_ = self._definition['format']
        if format_ in self._custom_formats:
            custom_format = self._custom_formats[format_]
            if isinstance(custom_format, str):
                self._generate_format(format_, format_ + '_re_pattern', custom_format)
            else:
                with self.l('if not custom_formats["{}"]({variable}):', format_):
                    self.exc('{name} must be {}', format_, rule='format')
        elif format_ in self.FORMAT_REGEXS:
            format_regex = self.FORMAT_REGEXS[format_]
            self._generate_format(format_, format_ + '_re_pattern', format_regex)
        elif format_ == 'regex':
            with self.l('try:', optimize=False):
                self.l('re.compile({variable})')
            with self.l('except Exception:'):
                self.exc('{name} must be a valid regex', rule='format')
        else:
            raise JsonSchemaDefinitionException('Unknown format: {}'.format(format_))