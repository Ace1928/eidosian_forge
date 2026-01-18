import copy
import sys
from alembic import command as alembic_command
from alembic import script as alembic_script
from alembic import util as alembic_util
from oslo_config import cfg
from oslo_log import log
import pbr.version
from keystone.common import sql
from keystone.common.sql import upgrades
import keystone.conf
from keystone.i18n import _
def do_upgrade(config, cmd):
    branch = None
    if (CONF.command.revision or CONF.command.delta) and (CONF.command.expand or CONF.command.contract):
        msg = _('Phase upgrade options do not accept revision specification')
        raise SystemExit(msg)
    if CONF.command.expand:
        branch = upgrades.EXPAND_BRANCH
        revision = f'{upgrades.EXPAND_BRANCH}@head'
    elif CONF.command.contract:
        branch = upgrades.CONTRACT_BRANCH
        revision = f'{upgrades.CONTRACT_BRANCH}@head'
    elif not CONF.command.revision and (not CONF.command.delta):
        msg = _('You must provide a revision or relative delta')
        raise SystemExit(msg)
    else:
        revision = CONF.command.revision or ''
        if '-' in revision:
            msg = _('Negative relative revision (downgrade) not supported')
            raise SystemExit(msg)
        delta = CONF.command.delta
        if delta:
            if '+' in revision:
                msg = _('Use either --delta or relative revision, not both')
                raise SystemExit(msg)
            if delta < 0:
                msg = _('Negative delta (downgrade) not supported')
                raise SystemExit(msg)
            revision = '%s+%d' % (revision, delta)
        if revision == 'head':
            revision = 'heads'
    if revision in upgrades.MILESTONES:
        expand_revisions = _find_milestone_revisions(config, revision, upgrades.EXPAND_BRANCH)
        contract_revisions = _find_milestone_revisions(config, revision, upgrades.CONTRACT_BRANCH)
        revisions = expand_revisions + contract_revisions
    else:
        revisions = [(revision, branch)]
    for revision, branch in revisions:
        do_alembic_command(config, cmd, revision=revision, sql=CONF.command.sql)