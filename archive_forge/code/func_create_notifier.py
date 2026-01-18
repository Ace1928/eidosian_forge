import os
import sys
def create_notifier(conf, log):
    if oslo_messaging and conf.get('use_oslo_messaging'):
        transport = oslo_messaging.get_notification_transport(conf.oslo_conf_obj, url=conf.get('transport_url'))
        notifier = oslo_messaging.Notifier(transport, os.path.basename(sys.argv[0]), driver=conf.get('driver'), topics=conf.get('topics'))
        return _MessagingNotifier(notifier)
    else:
        return _LogNotifier(log)