from django.core.management.base import BaseCommand
def output_hash(self, user_settings, default_settings, **options):
    output = []
    for key in sorted(user_settings):
        if key not in default_settings:
            output.append('%s = %s  ###' % (key, user_settings[key]))
        elif user_settings[key] != default_settings[key]:
            output.append('%s = %s' % (key, user_settings[key]))
        elif options['all']:
            output.append('### %s = %s' % (key, user_settings[key]))
    return output