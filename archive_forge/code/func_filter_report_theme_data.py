from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
def filter_report_theme_data(json):
    option_list = ['bullet_list_style', 'column_count', 'default_html_style', 'default_pdf_style', 'graph_chart_style', 'heading1_style', 'heading2_style', 'heading3_style', 'heading4_style', 'hline_style', 'image_style', 'name', 'normal_text_style', 'numbered_list_style', 'page_footer_style', 'page_header_style', 'page_orient', 'page_style', 'report_subtitle_style', 'report_title_style', 'table_chart_caption_style', 'table_chart_even_row_style', 'table_chart_head_style', 'table_chart_odd_row_style', 'table_chart_style', 'toc_heading1_style', 'toc_heading2_style', 'toc_heading3_style', 'toc_heading4_style', 'toc_title_style']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary