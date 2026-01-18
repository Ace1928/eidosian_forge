import re
from lxml import etree
def parse_error_message(message):
    try:
        if message.startswith('<html>'):
            message_tree = etree.XML(message)
            error_tag = message_tree.find('.//h3')
            if error_tag:
                error = error_tag.xpath('string()')
        elif message.startswith('<!DOCTYPE'):
            message_tree = etree.XML(message)
            error_tag = message_tree.find('.//title')
            if error_tag and 'Not Found' in error_tag.xpath('string()'):
                error = error_tag.xpath('string()')
            else:
                error_tag = message_tree.find('.//h1')
                if error_tag:
                    error = error_tag.xpath('string()')
        else:
            error = message
    except Exception:
        error = message
    finally:
        return error