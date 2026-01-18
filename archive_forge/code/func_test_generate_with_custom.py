import os
import unittest
from gae_ext_runtime import testutil
def test_generate_with_custom(self):
    self.write_file('index.rb', 'class Index; end')
    self.write_file('Gemfile', 'source "https://rubygems.org"')
    self.write_file('config.ru', 'run Index.app')
    unstub = self.stub_response('bundle exec rackup -p $PORT -E deployment')
    self.generate_configs(custom=True)
    unstub()
    app_yaml = self.file_contents('app.yaml')
    self.assertIn('runtime: custom\n', app_yaml)
    self.assertIn('env: flex\n', app_yaml)
    self.assertIn('entrypoint: bundle exec rackup -p $PORT -E deployment\n', app_yaml)
    dockerfile = self.file_contents('Dockerfile')
    self.assertEqual(dockerfile, DOCKERFILE_TEXT.format(ruby_version='', entrypoint='bundle exec rackup -p $PORT -E deployment'))
    dockerignore = self.file_contents('.dockerignore')
    self.assertIn('.dockerignore\n', dockerignore)
    self.assertIn('Dockerfile\n', dockerignore)
    self.assertIn('.git\n', dockerignore)
    self.assertIn('.hg\n', dockerignore)
    self.assertIn('.svn\n', dockerignore)