from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def extract_cmd_names(help_output):
    cmds = []
    for line in help_output.split('\n'):
        if line.startswith(' '):
            continue
        cmd = line.split(' ')[0]
        if line:
            cmds.append(cmd)
    return cmds