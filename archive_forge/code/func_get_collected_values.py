from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def get_collected_values(self, field_name):
    """Return the collected values for the given field."""
    return self._values.get(field_name, [])