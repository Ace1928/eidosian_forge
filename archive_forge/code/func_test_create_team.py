from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Organization, Team
@pytest.mark.django_db
def test_create_team(run_module, admin_user):
    org = Organization.objects.create(name='foo')
    result = run_module('team', {'name': 'foo_team', 'description': 'fooin around', 'state': 'present', 'organization': 'foo'}, admin_user)
    team = Team.objects.filter(name='foo_team').first()
    result.pop('invocation')
    assert result == {'changed': True, 'name': 'foo_team', 'id': team.id if team else None}
    team = Team.objects.get(name='foo_team')
    assert team.description == 'fooin around'
    assert team.organization_id == org.id