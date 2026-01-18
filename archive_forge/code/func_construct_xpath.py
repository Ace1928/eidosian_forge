import html
from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...utils import is_bs4_available, logging, requires_backends
def construct_xpath(self, xpath_tags, xpath_subscripts):
    xpath = ''
    for tagname, subs in zip(xpath_tags, xpath_subscripts):
        xpath += f'/{tagname}'
        if subs != 0:
            xpath += f'[{subs}]'
    return xpath