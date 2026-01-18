import os
import unittest
from gae_ext_runtime import testutil
def test_generate_with_ruby_version_no_write(self):
    """Tests generate_config_data with .ruby-version file."""
    self.write_file('index.rb', 'class Index; end')
    self.write_file('Gemfile', 'source "https://rubygems.org"')
    self.write_file('config.ru', 'run Index.app')
    self.write_file('.ruby-version', '2.3.1\n')
    appinfo = testutil.AppInfoFake(entrypoint='bundle exec ruby index.rb $PORT', runtime='ruby', vm=True)
    cfg_files = self.generate_config_data(appinfo=appinfo, deploy=True)
    self.assertFalse(os.path.exists(self.full_path('app.yaml')))
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', DOCKERFILE_TEXT.format(ruby_version='2.3.1', entrypoint='bundle exec ruby index.rb $PORT'))
    self.assertIn('.dockerignore', [f.filename for f in cfg_files])
    dockerignore = [f.contents for f in cfg_files if f.filename == '.dockerignore'][0]
    self.assertIn('.dockerignore\n', dockerignore)
    self.assertIn('Dockerfile\n', dockerignore)
    self.assertIn('.git\n', dockerignore)
    self.assertIn('.hg\n', dockerignore)
    self.assertIn('.svn\n', dockerignore)