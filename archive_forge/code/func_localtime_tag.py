import zoneinfo
from datetime import datetime
from datetime import timezone as datetime_timezone
from datetime import tzinfo
from django.template import Library, Node, TemplateSyntaxError
from django.utils import timezone
@register.tag('localtime')
def localtime_tag(parser, token):
    """
    Force or prevent conversion of datetime objects to local time,
    regardless of the value of ``settings.USE_TZ``.

    Sample usage::

        {% localtime off %}{{ value_in_utc }}{% endlocaltime %}
    """
    bits = token.split_contents()
    if len(bits) == 1:
        use_tz = True
    elif len(bits) > 2 or bits[1] not in ('on', 'off'):
        raise TemplateSyntaxError("%r argument should be 'on' or 'off'" % bits[0])
    else:
        use_tz = bits[1] == 'on'
    nodelist = parser.parse(('endlocaltime',))
    parser.delete_first_token()
    return LocalTimeNode(nodelist, use_tz)