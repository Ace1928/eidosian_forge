import argparse
from transformers import ConvBertConfig, ConvBertModel, TFConvBertModel, load_tf_weights_in_convbert
from transformers.utils import logging
def convert_orig_tf1_checkpoint_to_pytorch(tf_checkpoint_path, convbert_config_file, pytorch_dump_path):
    conf = ConvBertConfig.from_json_file(convbert_config_file)
    model = ConvBertModel(conf)
    model = load_tf_weights_in_convbert(model, conf, tf_checkpoint_path)
    model.save_pretrained(pytorch_dump_path)
    tf_model = TFConvBertModel.from_pretrained(pytorch_dump_path, from_pt=True)
    tf_model.save_pretrained(pytorch_dump_path)