import html
from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...utils import is_bs4_available, logging, requires_backends
def get_three_from_single(self, html_string):
    html_code = BeautifulSoup(html_string, 'html.parser')
    all_doc_strings = []
    string2xtag_seq = []
    string2xsubs_seq = []
    for element in html_code.descendants:
        if isinstance(element, bs4.element.NavigableString):
            if type(element.parent) != bs4.element.Tag:
                continue
            text_in_this_tag = html.unescape(element).strip()
            if not text_in_this_tag:
                continue
            all_doc_strings.append(text_in_this_tag)
            xpath_tags, xpath_subscripts = self.xpath_soup(element)
            string2xtag_seq.append(xpath_tags)
            string2xsubs_seq.append(xpath_subscripts)
    if len(all_doc_strings) != len(string2xtag_seq):
        raise ValueError('Number of doc strings and xtags does not correspond')
    if len(all_doc_strings) != len(string2xsubs_seq):
        raise ValueError('Number of doc strings and xsubs does not correspond')
    return (all_doc_strings, string2xtag_seq, string2xsubs_seq)