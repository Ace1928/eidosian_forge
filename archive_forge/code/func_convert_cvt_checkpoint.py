import argparse
import json
from collections import OrderedDict
import torch
from huggingface_hub import cached_download, hf_hub_url
from transformers import AutoImageProcessor, CvtConfig, CvtForImageClassification
def convert_cvt_checkpoint(cvt_model, image_size, cvt_file_name, pytorch_dump_folder):
    """
    Fucntion to convert the microsoft cvt checkpoint to huggingface checkpoint
    """
    img_labels_file = 'imagenet-1k-id2label.json'
    num_labels = 1000
    repo_id = 'huggingface/label-files'
    num_labels = num_labels
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, img_labels_file, repo_type='dataset')), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    id2label = id2label
    label2id = {v: k for k, v in id2label.items()}
    config = config = CvtConfig(num_labels=num_labels, id2label=id2label, label2id=label2id)
    if cvt_model.rsplit('/', 1)[-1][4:6] == '13':
        config.depth = [1, 2, 10]
    elif cvt_model.rsplit('/', 1)[-1][4:6] == '21':
        config.depth = [1, 4, 16]
    else:
        config.depth = [2, 2, 20]
        config.num_heads = [3, 12, 16]
        config.embed_dim = [192, 768, 1024]
    model = CvtForImageClassification(config)
    image_processor = AutoImageProcessor.from_pretrained('facebook/convnext-base-224-22k-1k')
    image_processor.size['shortest_edge'] = image_size
    original_weights = torch.load(cvt_file_name, map_location=torch.device('cpu'))
    huggingface_weights = OrderedDict()
    list_of_state_dict = []
    for idx in range(len(config.depth)):
        if config.cls_token[idx]:
            list_of_state_dict = list_of_state_dict + cls_token(idx)
        list_of_state_dict = list_of_state_dict + embeddings(idx)
        for cnt in range(config.depth[idx]):
            list_of_state_dict = list_of_state_dict + attention(idx, cnt)
    list_of_state_dict = list_of_state_dict + final()
    for gg in list_of_state_dict:
        print(gg)
    for i in range(len(list_of_state_dict)):
        huggingface_weights[list_of_state_dict[i][0]] = original_weights[list_of_state_dict[i][1]]
    model.load_state_dict(huggingface_weights)
    model.save_pretrained(pytorch_dump_folder)
    image_processor.save_pretrained(pytorch_dump_folder)