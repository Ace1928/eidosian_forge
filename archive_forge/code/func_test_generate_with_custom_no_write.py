import os
import unittest
from gae_ext_runtime import testutil
def test_generate_with_custom_no_write(self):
    """Tests generate_config_data with custom=True.

        Tests that app.yaml is written with correct parameters and
        Dockerfile, .dockerignore contents are correctly returned by method.
        """
    self.write_file('index.rb', 'class Index; end')
    self.write_file('Gemfile', 'source "https://rubygems.org"')
    self.write_file('config.ru', 'run Index.app')
    unstub = self.stub_response('bundle exec rackup -p $PORT -E deployment')
    cfg_files = self.generate_config_data(custom=True)
    unstub()
    app_yaml = self.file_contents('app.yaml')
    self.assertIn('runtime: custom\n', app_yaml)
    self.assertIn('env: flex\n', app_yaml)
    self.assertIn('entrypoint: bundle exec rackup -p $PORT -E deployment\n', app_yaml)
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', DOCKERFILE_TEXT.format(ruby_version='', entrypoint='bundle exec rackup -p $PORT -E deployment'))
    self.assertIn('.dockerignore', [f.filename for f in cfg_files])
    dockerignore = [f.contents for f in cfg_files if f.filename == '.dockerignore'][0]
    self.assertIn('.dockerignore\n', dockerignore)
    self.assertIn('Dockerfile\n', dockerignore)
    self.assertIn('.git\n', dockerignore)
    self.assertIn('.hg\n', dockerignore)
    self.assertIn('.svn\n', dockerignore)