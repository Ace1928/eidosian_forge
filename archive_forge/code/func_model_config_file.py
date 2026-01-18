from ._base import *
@property
def model_config_file(self):
    if not self.model_dir:
        return None
    if not self._model_config_file:
        for fname in ['config.json', 'model_config.json', 'model.json']:
            config_path = File.join(self.model_dir, fname)
            if File.exists(config_path):
                self._model_config_file = config_path
                return self._model_config_file
    self._model_config_file = File.join(self.model_dir, 'config.json')
    return self._model_config_file