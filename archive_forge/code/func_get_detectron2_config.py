from ...configuration_utils import PretrainedConfig
from ...utils import is_detectron2_available, logging
def get_detectron2_config(self):
    detectron2_config = detectron2.config.get_cfg()
    for k, v in self.detectron2_config_args.items():
        attributes = k.split('.')
        to_set = detectron2_config
        for attribute in attributes[:-1]:
            to_set = getattr(to_set, attribute)
        setattr(to_set, attributes[-1], v)
    return detectron2_config