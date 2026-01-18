import warnings
from bleach import ALLOWED_ATTRIBUTES, ALLOWED_TAGS, clean
from traitlets import Any, Bool, List, Set, Unicode
from .base import Preprocessor
def sanitize_code_outputs(self, outputs):
    """
        Sanitize code cell outputs.

        Removes 'text/javascript' fields from display_data outputs, and
        runs `sanitize_html_tags` over 'text/html'.
        """
    for output in outputs:
        if output['output_type'] in ('stream', 'error'):
            continue
        data = output.data
        to_remove = []
        for key in data:
            if key in self.safe_output_keys:
                continue
            if key in self.sanitized_output_types:
                self.log.info('Sanitizing %s', key)
                data[key] = self.sanitize_html_tags(data[key])
            else:
                to_remove.append(key)
        for key in to_remove:
            self.log.info('Removing %s', key)
            del data[key]
    return outputs