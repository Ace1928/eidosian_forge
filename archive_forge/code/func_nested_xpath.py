from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def nested_xpath(self, xpath, **context):
    """
        Create a nested loader with an xpath selector.
        The supplied selector is applied relative to selector associated
        with this :class:`ItemLoader`. The nested loader shares the item
        with the parent :class:`ItemLoader` so calls to :meth:`add_xpath`,
        :meth:`add_value`, :meth:`replace_value`, etc. will behave as expected.
        """
    selector = self.selector.xpath(xpath)
    context.update(selector=selector)
    subloader = self.__class__(item=self.item, parent=self, **context)
    return subloader