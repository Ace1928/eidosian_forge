from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def build_payload_for_bearychat_attachment(module, title, text, color, images):
    attachment = {}
    if title is not None:
        attachment['title'] = title
    if text is not None:
        attachment['text'] = text
    if color is not None:
        attachment['color'] = color
    if images is not None:
        target_images = attachment.setdefault('images', [])
        if not isinstance(images, (list, tuple)):
            images = [images]
        for image in images:
            if isinstance(image, dict) and 'url' in image:
                image = {'url': image['url']}
            elif hasattr(image, 'startswith') and image.startswith('http'):
                image = {'url': image}
            else:
                module.fail_json(msg="BearyChat doesn't have support for this kind of attachment image")
            target_images.append(image)
    return attachment