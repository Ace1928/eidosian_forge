from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def parse_xml_to_dict(self, xmldata):
    """Parse raw XML from system-cli and create an Ansible parseable dictonary"""
    xml_import_ok = True
    xml_parse_ok = True
    importing = 'ast'
    try:
        import ast
        importing = 'xml.parsers.expat'
        import xml.parsers.expat
    except ImportError:
        self.result_dict['status'] = 'XML parsing failed. Cannot import %s!' % importing
        self.result_dict['stdout'] = str(xmldata)
        self.result_dict['result_value'] = -1
        xml_import_ok = False
    if xml_import_ok:
        xml_str = xmldata.decode('utf-8').replace('\n', '---')
        xml_parser = xml.parsers.expat.ParserCreate()
        xml_parser.StartElementHandler = self._start_element
        xml_parser.CharacterDataHandler = self._char_data
        xml_parser.EndElementHandler = self._end_element
        try:
            xml_parser.Parse(xml_str)
        except xml.parsers.expat.ExpatError as errcode:
            self.result_dict['status'] = 'XML parsing failed: ' + str(errcode)
            self.result_dict['stdout'] = str(xmldata)
            self.result_dict['result_value'] = -1
            xml_parse_ok = False
        if xml_parse_ok:
            self.result_dict['status'] = self.result_dict['xml_dict']['results']['attrs']['status']
            stdout_string = self._format_escaped_data(self.result_dict['xml_dict']['cli-output']['data'])
            self.result_dict['stdout'] = stdout_string
            for line in stdout_string.split('\n'):
                stripped_line = line.strip()
                if len(stripped_line) > 1:
                    self.result_dict['stdout_lines'].append(stripped_line)
                    if self.exclude_lines:
                        if self.include_lines in stripped_line and self.exclude_lines not in stripped_line:
                            self.result_dict['stdout_lines_filter'].append(stripped_line)
                    elif self.include_lines and self.include_lines in stripped_line:
                        self.result_dict['stdout_lines_filter'].append(stripped_line)
            self.result_dict['xml_dict']['cli-output']['data'] = stdout_string
            cli_result_value = self.result_dict['xml_dict']['cli-result-value']['data']
            try:
                cli_result_value = ast.literal_eval(cli_result_value)
            except (SyntaxError, ValueError):
                pass
            try:
                self.result_dict['result_value'] = int(cli_result_value)
            except ValueError:
                self.result_dict['result_value'] = cli_result_value
    return self.result_dict