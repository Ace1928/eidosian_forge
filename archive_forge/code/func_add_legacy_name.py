from django.db import migrations, models
def add_legacy_name(apps, schema_editor):
    alias = schema_editor.connection.alias
    ContentType = apps.get_model('contenttypes', 'ContentType')
    for ct in ContentType.objects.using(alias):
        try:
            ct.name = apps.get_model(ct.app_label, ct.model)._meta.object_name
        except LookupError:
            ct.name = ct.model
        ct.save()