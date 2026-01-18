import io
import os
from tensorflow.lite.toco.logging import toco_conversion_log_pb2 as _toco_conversion_log_pb2
from tensorflow.python.lib.io import file_io as _file_io
from tensorflow.python.platform import resource_loader as _resource_loader
def gen_conversion_log_html(conversion_log_dir, quantization_enabled, tflite_graph_path):
    """Generates an HTML report about the conversion process.

  Args:
    conversion_log_dir: A string specifying the file directory of the conversion
      logs. It's required that before calling this function, the
      `conversion_log_dir`
      already contains the following files: `toco_log_before.pb`,
        `toco_log_after.pb`, `toco_tf_graph.dot`,
        `toco_tflite_graph.dot`.
    quantization_enabled: A boolean, passed from the tflite converter to
      indicate whether post-training quantization is enabled during conversion.
    tflite_graph_path: A string, the filepath to the converted TFLite model.

  Raises:
    IOError: When any of the required files doesn't exist.
  """
    template_filename = _resource_loader.get_path_to_datafile('template.html')
    if not os.path.exists(template_filename):
        raise IOError("Failed to generate HTML: file '{0}' doesn't exist.".format(template_filename))
    toco_log_before_path = os.path.join(conversion_log_dir, 'toco_log_before.pb')
    toco_log_after_path = os.path.join(conversion_log_dir, 'toco_log_after.pb')
    dot_before_path = os.path.join(conversion_log_dir, 'toco_tf_graph.dot')
    dot_after_path = os.path.join(conversion_log_dir, 'toco_tflite_graph.dot')
    if not os.path.exists(toco_log_before_path):
        raise IOError("Failed to generate HTML: file '{0}' doesn't exist.".format(toco_log_before_path))
    if not os.path.exists(toco_log_after_path):
        raise IOError("Failed to generate HTML: file '{0}' doesn't exist.".format(toco_log_after_path))
    if not os.path.exists(dot_before_path):
        raise IOError("Failed to generate HTML: file '{0}' doesn't exist.".format(dot_before_path))
    if not os.path.exists(dot_after_path):
        raise IOError("Failed to generate HTML: file '{0}' doesn't exist.".format(dot_after_path))
    html_generator = HTMLGenerator(template_filename, os.path.join(conversion_log_dir, 'toco_conversion_summary.html'))
    toco_conversion_log_before = _toco_conversion_log_pb2.TocoConversionLog()
    toco_conversion_log_after = _toco_conversion_log_pb2.TocoConversionLog()
    with open(toco_log_before_path, 'rb') as f:
        toco_conversion_log_before.ParseFromString(f.read())
    with open(toco_log_after_path, 'rb') as f:
        toco_conversion_log_after.ParseFromString(f.read())
    with io.open(dot_before_path, 'r', encoding='utf-8') as f:
        dot_before = f.read().rstrip()
    with io.open(dot_after_path, 'r', encoding='utf-8') as f:
        dot_after = f.read().rstrip()
    html_generator.generate(toco_conversion_log_before, toco_conversion_log_after, quantization_enabled, dot_before, dot_after, toco_conversion_log_after.toco_err_logs, tflite_graph_path)