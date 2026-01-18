import copy
from datetime import datetime
from typing import Callable, Dict, Optional, Union
from packaging import version
import wandb
from wandb.sdk.lib import telemetry
@torch.no_grad()
def on_fit_epoch_end(self, trainer: DetectionTrainer):
    if self.task in self.supported_tasks and self.train_epoch != trainer.epoch:
        self.train_epoch = trainer.epoch
        if (self.train_epoch + 1) % self.epoch_logging_interval == 0:
            validator = trainer.validator
            dataloader = validator.dataloader
            class_label_map = validator.names
            self.device = next(trainer.model.parameters()).device
            if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel):
                model = trainer.model.module
            else:
                model = trainer.model
            self.model = copy.deepcopy(model).eval().to(self.device)
            self.predictor.setup_model(model=self.model, verbose=False)
            if self.task == 'pose':
                self.train_validation_table = plot_pose_validation_results(dataloader=dataloader, class_label_map=class_label_map, model_name=self.model_name, predictor=self.predictor, visualize_skeleton=self.visualize_skeleton, table=self.train_validation_table, max_validation_batches=self.max_validation_batches, epoch=trainer.epoch)
            elif self.task == 'segment':
                self.train_validation_table = plot_segmentation_validation_results(dataloader=dataloader, class_label_map=class_label_map, model_name=self.model_name, predictor=self.predictor, table=self.train_validation_table, max_validation_batches=self.max_validation_batches, epoch=trainer.epoch)
            elif self.task == 'detect':
                self.train_validation_table = plot_detection_validation_results(dataloader=dataloader, class_label_map=class_label_map, model_name=self.model_name, predictor=self.predictor, table=self.train_validation_table, max_validation_batches=self.max_validation_batches, epoch=trainer.epoch)
            elif self.task == 'classify':
                self.train_validation_table = plot_classification_validation_results(dataloader=dataloader, model_name=self.model_name, predictor=self.predictor, table=self.train_validation_table, max_validation_batches=self.max_validation_batches, epoch=trainer.epoch)
        if self.enable_model_checkpointing:
            self._save_model(trainer)
        self.model.to('cpu')
        trainer.model.to(self.device)