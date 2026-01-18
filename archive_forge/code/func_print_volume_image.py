import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def print_volume_image(image_resp_tuple):
    image = image_resp_tuple[1]
    vt = image['os-volume_upload_image'].get('volume_type')
    if vt is not None:
        image['os-volume_upload_image']['volume_type'] = vt.get('name')
    print_dict(image['os-volume_upload_image'])