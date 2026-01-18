import abc
import collections
import json
from tensorboard.uploader import util
def format_experiment(self, experiment, experiment_url):
    data = [('url', experiment_url), ('name', experiment.name), ('description', experiment.description), ('id', experiment.experiment_id), ('created', util.format_time_absolute(experiment.create_time)), ('updated', util.format_time_absolute(experiment.update_time)), ('runs', experiment.num_runs), ('tags', experiment.num_tags), ('scalars', experiment.num_scalars), ('tensor_bytes', experiment.total_tensor_bytes), ('binary_object_bytes', experiment.total_blob_bytes)]
    return json.dumps(collections.OrderedDict(data), indent=self._JSON_INDENT)