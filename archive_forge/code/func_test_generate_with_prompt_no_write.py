import os
import unittest
from gae_ext_runtime import testutil
def test_generate_with_prompt_no_write(self):
    """Tests generate_config_data with entrypoint given by prompt."""
    self.write_file('index.rb', 'class Index; end')
    self.write_file('Gemfile', 'source "https://rubygems.org"')
    unstub = self.stub_response('bundle exec ruby index.rb $PORT')
    cfg_files = self.generate_config_data(deploy=True)
    unstub()
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', DOCKERFILE_TEXT.format(ruby_version='', entrypoint='bundle exec ruby index.rb $PORT'))
    self.assertIn('.dockerignore', [f.filename for f in cfg_files])
    dockerignore = [f.contents for f in cfg_files if f.filename == '.dockerignore'][0]
    self.assertIn('.dockerignore\n', dockerignore)
    self.assertIn('Dockerfile\n', dockerignore)
    self.assertIn('.git\n', dockerignore)
    self.assertIn('.hg\n', dockerignore)
    self.assertIn('.svn\n', dockerignore)