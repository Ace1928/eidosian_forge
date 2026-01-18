from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import missing_required_lib
import traceback
import sys
import os
def control_xml_to_inifile_params(filepath, module):
    if not HAS_BS4_LIBRARY:
        module.fail_json(msg=missing_required_lib('bs4'), exception=BS4_LIBRARY_IMPORT_ERROR)
    infile = open(filepath + '/control_utf8.xml', 'r')
    contents = infile.read()
    soup = BeautifulSoup(markup=contents, features='lxml-xml')
    space = soup.find('components')
    component_list = space.findChildren('component', recursive=False)
    inifile_output = open('generated_inifile_params', 'w')
    inifile_params_header = "############\n    # SWPM Unattended Parameters inifile.params generated export\n    #\n    #\n    # Export of all SWPM Component and the SWPM Unattended Parameters. Not all components have SWPM Unattended Parameters.\n    #\n    # All parameters are commented-out, each hash # before the parameter is removed to activate the parameter.\n    # When running SWPM in Unattended Mode, the activated parameters will create a new SWPM file in the sapinst directory.\n    # If any parameter is marked as 'encode', the plaintext value will be coverted to DES hash\n    # for this parameter in the new SWPM file (in the sapinst directory).\n    #\n    # An inifile.params is otherwise obtained after running SWPM as GUI or Unattended install,\n    # and will be generated for a specific Product ID (such as 'NW_ABAP_OneHost:S4HANA1809.CORE.HDB.CP').\n    ############\n\n\n\n    ############\n    # MANUAL\n    ############\n\n    # The folder containing all archives that have been downloaded from http://support.sap.com/swdc and are supposed to be used in this procedure\n    # archives.downloadBasket =\n    "
    inifile_output.write(inifile_params_header)
    for component in component_list:
        component_key_name_text = component['name']
        component_key_display_name = component.find('display-name')
        if component_key_display_name is not None:
            component_key_display_name_text = component_key_display_name.get_text()
        inifile_output.write('\n\n\n\n############\n# Component: %s\n# Component Display Name: %s\n############\n' % (component_key_name_text, component_key_display_name_text))
        for parameter in component.findChildren('parameter'):
            component_parameter_key_encode = parameter.get('encode', None)
            component_parameter_key_inifile_name = parameter.get('defval-for-inifile-generation', None)
            component_parameter_key_defval = parameter.get('defval', '')
            component_parameter_contents_doclong_text = parameter.get_text().replace('\n', '')
            if component_parameter_key_inifile_name is not None:
                inifile_output.write('\n# %s' % component_parameter_contents_doclong_text)
                if component_parameter_key_encode == 'true':
                    inifile_output.write('\n# Encoded parameter. Plaintext values will be coverted to DES hash')
                inifile_output.write('\n# %s = %s\n' % (component_parameter_key_inifile_name, component_parameter_key_defval))
    inifile_output.close()