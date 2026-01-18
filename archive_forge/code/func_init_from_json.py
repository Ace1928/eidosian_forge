from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, Union
from wandb import util
@staticmethod
def init_from_json(json_obj: dict, source_artifact: 'Artifact') -> Optional['WBValue']:
    """Initialize a `WBValue` from a JSON blob based on the class that creatd it.

        Looks through all subclasses and tries to match the json obj with the class
        which created it. It will then call that subclass' `from_json` method.
        Importantly, this function will set the return object's `source_artifact`
        attribute to the passed in source artifact. This is critical for artifact
        bookkeeping. If you choose to create a wandb.Value via it's `from_json` method,
        make sure to properly set this `artifact_source` to avoid data duplication.

        Args:
            json_obj (dict): A JSON dictionary to deserialize. It must contain a `_type`
                key. This is used to lookup the correct subclass to use.
            source_artifact (wandb.Artifact): An artifact which will hold any additional
                resources which were stored during the `to_json` function.

        Returns:
            wandb.Value: a newly created instance of a subclass of wandb.Value
        """
    class_option = WBValue.type_mapping().get(json_obj['_type'])
    if class_option is not None:
        obj = class_option.from_json(json_obj, source_artifact)
        obj._set_artifact_source(source_artifact)
        return obj
    return None