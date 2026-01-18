from setuptools.command.develop import develop as old_develop
def install_for_development(self):
    self.reinitialize_command('build_src', inplace=1)
    self.run_command('build_scripts')
    old_develop.install_for_development(self)