from .. import config
def get_sections(self):
    return [(self, GitConfigSectionDefault('default', self._config))]